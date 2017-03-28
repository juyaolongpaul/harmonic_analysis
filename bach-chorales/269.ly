\version "2.10.33"

\header {
  title = "269 - Jesu, der du meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    d4 d4 a4 bes4 
    c4 bes8 a8 a2 
    g2 bes4 bes4 
    a4 g4 fis4 g4 
    a1 }  %5
  a4 bes4 c4 a4 
  bes4 a8 g8 g2 
  f2 bes4 c4 
  d4 bes4 ees4 d4 
  c2 bes2   %10
  a4 bes4 c4 d4 
  c4 bes4 a2 
  g4 a4 bes4 c8 bes8 
  a4. g8 g2 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    g8 fis8 g4 fis4 g4 
    g8 fis8 g4 g4 fis4 
    d2 g4 g8 f8 
    ees8 d8 d8 c8 c4 d4 
    d1 }  %5
  fis4 g4 g4 f4 
  f4 f4 f4 e4 
  c2 f4 f4 
  f8 ees8 f8 d8 g4 g4 
  g4 f8 ees8 d2   %10
  f4 ees8 d8 c4 f8 g8 
  a4 g4 fis2 
  g4 d4 g4 g4 
  g4 fis4 d2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    bes8 c8 d4 d4 d4 
    c4 d4 ees4 d8 c8 
    bes2 d4 g,4 
    g8 fis8 g4 a8 c8 bes8 g8 
    fis1 }  %5
  d'4 d4 c4 c4 
  bes8 c8 d4 c4. bes8 
  a2 d4 a4 
  bes4 bes4 bes8 c8 d8 bes8 
  g8 ees'8 a,4 bes2   %10
  c4 bes4 f'8 ees8 d4 
  d4 d4 d2 
  d4 d4 d4 c8 d8 
  ees8 c8 a8 d8 b2 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    g8 a8 bes8 c8 d4 c8 bes8 
    a4 g4 c,4 d4 
    g,2 g'8 f8 ees8 d8 
    c8 d8 ees4 a,4 g4 
    d'1 }  %5
  d4 g8 f8 e4 f8 e8 
  d4 c8 bes8 c2 
  f,2 d'4 f4 
  bes8 c8 d8 bes8 g8 a8 bes8 g8 
  ees8 c8 f4 bes,2   %10
  f'4 g4 a4 bes8 a8 
  g8 fis8 g4 d2 
  bes'8 a8 g8 fis8 g8 f8 ees8 d8 
  c8 a8 d4 g,2 
  
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