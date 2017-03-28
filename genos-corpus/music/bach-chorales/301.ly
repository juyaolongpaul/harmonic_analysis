\version "2.10.33"

\header {
  title = "301 - Ach, lieben Christen, seid getrost"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 bes4 
    bes4 g8 a8 bes8 c8 d4 
    c4 c4 bes4 bes4 
    a4 g8 a8 bes4 c8 bes8 
    a2 g4 } c4 
  d4 d4 a8 bes8 c4   %5
  bes4 bes4 a4 bes4 
  c4 g8 a8 bes4 a4 
  g4 g4 f4 bes4 
  a4 g8 a8 bes4 c8 bes8 
  a2 g4   %10
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    f4 ees4 d4 d4 
    ees4 ees16 d16 ees8 d4 d4 
    d8 c'8 bes8 a8 g4 g8 f8 
    ees4 d8 c8 bes4 } f'4 
  f4 f4 c4. a'8   %5
  a8 g16 fis16 g4 fis4 g4 
  g8 f8 ees4 d8 e8 f4 
  f4 e4 c4 d4 
  c8 d8 e8 fis8 g4 g4 
  g4 fis4 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    d8 c8 bes4 bes4 f4 
    g4 a4 f4 g4 
    fis8 ees'8 d8 c8 bes8 a8 g4 
    g4 fis4 g4 } f4 
  bes4 bes4 f'4 fis8 a,8   %5
  bes8 c8 d4 d4 d4 
  g,4 c4 bes4 c4 
  c8 g8 c8 bes8 a4 f8 g8 
  a8 bes8 c4 d4 c8 d8 
  ees8 a,8 d8 c8 b4   %10
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    d4 ees8 f8 g8 a8 bes4 
    ees,4 f4 bes,4 g4 
    d'4 e8 fis8 g8 f8 ees8 d8 
    c4 d4 g,4 } a4 
  bes8 c8 d8 ees8 f8 g8 a8 fis8   %5
  g8 a8 bes8 c8 d4 g,8 f8 
  ees8 d8 c4 g4 a8 bes8 
  c8 bes8 c4 f,4 d'8 e8 
  f4 c4 g'8 f8 ees8 d8 
  c4 d4 g,4   %10
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}