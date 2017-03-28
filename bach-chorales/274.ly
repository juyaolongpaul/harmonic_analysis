\version "2.10.33"

\header {
  title = "274 - O Ewigkeit, du Donnerwort"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c' {
  \repeat volta 2 {
    f4. g8 a4 bes4 
    c4 c4 d4 e4 
    f2 f,4. g8 
    a4 bes4 c4 c4 
    bes4 a4 g2     %5
    a2 f4 f4 
    bes4 a4 g2 
    f1 }
  c'2 g4 a4 
  bes4 bes4 a4. g8   %10
  g2 c2 
  d4 e4 f4 a,4 
  g4. g8 f2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    c4 d8 e8 f4 f4 
    f8 g8 a4 a8 g8 bes8 a8 
    f2 d4. e8 
    f4 g4 fis8 g8 a4 
    a8 g8 g8 f8 c2     %5
    c2 d8 e8 f4 
    f8 e8 f4 f4 e4 
    c1 }
  f2 f8 e8 ees8 d8 
  d4 g4 g4 fis4   %10
  d2 f4. ees8 
  d8 f8 bes4 a8 g8 f4 
  f4 e4 c2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    a4 bes4 c4 d4 
    c4 f4 f8 g8 g8 cis,8 
    d2 a4 bes4 
    f4 d'4 ees4 d4 
    d4 c8 d8 e2     %5
    f4 ees4 d4 d8 c8 
    bes8 c8 d4 d4 c8 bes8 
    a1 }
  a8 bes8 c2 c4 
  bes4 bes4 ees4 d8. c16   %10
  bes2 a8 c8 f4~ 
  f4 g4 f8 e8 d4 
  d4 c8 bes8 a2 
  
}

baixo = \relative c, {
  \repeat volta 2 {
    f4 f'2 d4 
    a4 a'4 b4 cis8 a8 
    d2 d,2 
    d'4 c8 bes8 a8 g8 fis8 d8 
    g4 a8 bes8 c2     %5
    f,2 bes8 c8 bes8 a8 
    g4 d8 c8 bes4 c4 
    f,1 }
  a'4. bes8 c4 fis,4 
  g8 f8 ees8 d8 c4 d4   %10
  g,2 a2 
  bes8 bes'8 a8 g8 d'8 d,8 d'8 c8 
  bes8 g8 c8 c,8 f2 
  
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