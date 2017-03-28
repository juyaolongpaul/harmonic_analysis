\version "2.10.33"

\header {
  title = "50 - In allen meinen Taten"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c'' {
  \partial 4 a4 
    f4 g4 a8 bes8 c4 
  bes2 a4 a8 bes8 
  c4 c4 g4 a8 g8 
  f2 e4 c4 
  f4 g4 a8 bes16 c16 bes8 a8   %5
  g2. a4 
  f4 g4 a8 bes8 c4 
  bes2 a4 a8 bes8 
  c4 c4 g4 a8 g8 
  f2 e4 c4   %10
  f4 g4 a4 g8 a16 bes16 
  a4 g4 f
  
}

alto = \relative c' {
  \partial 4 f4 
    d4 e4 f8 g8 a4 
  a4 g4 f4 f4 
  f8 e8 f8 g8 e8 d8 c4 
  c4 b4 c4 e4 
  d4 c4 c4 d4   %5
  e2. e4 
  d4 c4 c4 d4 
  d4 g4 fis4 f4 
  f4 f4 e8 d8 c4 
  c4 b4 c4 g4   %10
  c4 bes4 a8 c8 f4 
  f4 e4 c 
  
}

tenor = \relative c' {
  \partial 4 c4 
    d4 c8 bes8 a4. g8 
  f4 c'4 c4 c4 
  c4 c4 c4 c8 b8 
  a8 g8 f4 g4 bes4 
  a4 g4 f4 f4   %5
  c'2. a4 
  a4 g4 f8 g8 a4 
  g8 a8 bes8 c8 d4 d4 
  c8 bes8 c8 d8 e4 f4 
  f,8 a8 g8 f8 g4 e4   %10
  f8 c'8 d8 e8 f8 c8 bes8 f8 
  c'4. bes8 a4
  
}

baixo = \relative c {
  \partial 4 f4 
    bes4 a8 g8 f4. e8 
  d4 e4 f4 f8 g8 
  a4 a,8 bes8 c4 a4 
  d2 c4 c4 
  d4 e4 f4 bes,4   %5
  c2. cis4 
  d4 e4 f4 fis4 
  g4 g,4 d'4 d'4 
  a8 g8 a8 bes8 c4 f,8 e8 
  d8 c8 d4 c4 bes4   %10
  a4 g4 f4 d'4 
  c8 bes8 c4 f,
  
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