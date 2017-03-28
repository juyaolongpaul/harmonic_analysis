\version "2.10.33"

\header {
  title = "225 - Gott, der du selber bist das Licht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    g,4 a8 bes8 c4 bes4 
    a4 a4 g4 c4 
    c4 c4 d4 c4 
    d4 e4 f4 d4 
    c4 bes4 a4 g4     %5
    a2 g4 } a4 
  bes4 c4 d4 ees4 
  d4 c4 bes4 d4 
  f4 ees4 d4 g,8 a8 
  bes4 a g   %10
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g4 fis8 g8 a4. g8~ 
    g8 fis16 e16 fis4 d4 g4 
    f8 e8 f4 f4 f4 
    bes8 d4 cis8 d4 bes4 
    a4 d,8 e8 fis4 g4     %5
    g4 fis4 d4 } f4 
  f8 bes4 a8 bes4 c4~ 
  c8 bes4 a8 f4 f4 
  f8 a8 g8 f8 f4 ees4 
  d8 g4 fis8 d4   %10
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 bes8  c8 
    d4 c8 d8 ees8 d8 d4 
    ees8 c8 a8 d16 c16 bes4 c8 bes8 
    a8 bes8 c8 a8 bes4 c4 
    bes8 a8 g4 a4 g'4~ 
    g8 fis8 g4 d8 c8 bes4     %5
    ees8 c8 a8 d16 c16 bes4 } c4 
  d4 ees4 f4 f4 
  f4 g8 f16 ees16 d4 d4 
  c8 d8 bes8 c8 d4 c8 fis,8 
  g8 d'8 d8. c16 b4   %10
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 g8  a8 
    bes4 a8 g8 fis4 g4 
    c8 a8 d4 g,4 e4 
    f8 g8 a8 f8 bes8 bes'4 a8 
    g8 f8 e4 d4 g,4 
    a4 bes8 c8 d4 ees8 d8     %5
    c4 d4 g,4 } f'8 ees8 
  d4 c4 bes8 bes'4 a8 
  bes8 d,8 ees8 f8 bes,4 bes4 
  a8 f8 g8 a8 bes4 c4 
  bes8 g8 d'4 g, %10
  
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